//
//  jhDrawAxisLayer.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

class jhDrawAxisLayer<T> : CALayer {
    
    private let panelID: Int
    
    private var mContext: CGContext?
    private var horizontalGuide: Bool
    
    /// Axes
    private var mUnitOfHorizontalAxes : CGFloat = 100
    private var mcountOfHorizontalAxes : Int = 3
    private var mMargin : CGFloat
    private var maxY : CGFloat
    
    internal var mLineWidth : CGFloat = 1
    internal var mColor : CGColor = UIColor(red: 229, green: 229, blue: 229).cgColor

    var axisDistance : CGFloat
    
    init(_ x: jhPanel<T>, layer: Any, panelID: Int, hGuide: Bool, countVaxis: Int, maxY: CGFloat) {
        self.mContext = UIGraphicsGetCurrentContext()
        self.axisDistance = x.axisDistance
        self.mcountOfHorizontalAxes = countVaxis

        self.panelID = panelID
        self.mMargin = jhGS.s.jhAMarginCommonV
        self.horizontalGuide = hGuide
        self.maxY = maxY
        
        super.init(layer: layer)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(in ctx: CGContext) {
        
        mContext = ctx
        
        var xlocation : CGFloat = 0
        
        for x in 1..<jhDataCenter2.mCountOfaxes_view+1 {
            
            xlocation = CGFloat(x) * axisDistance + mMargin
            drawLine(xlocation, mMargin, xlocation, jhDraw.ARQ-mMargin)
            //TODO: LABEL
            //            self.contents = (drawText(str: String(x), x: xlocation-10, y: jhDraw.maxR-mMargin, width: 10, height: 10)).cgImage
            self.addSublayer(drawText(str: String(x), x: xlocation-jhGS.s.jhATextPanelSize, y: jhDraw.ARQ-mMargin, width: jhGS.s.jhATextPanelSize, height: jhGS.s.jhATextPanelSize))
        }
        
        /// horizontal axes + left size text
        let temp : Int = mcountOfHorizontalAxes + 1
        let intervalVaxis : Int = Int(maxY) / temp
        
        for x in 1..<temp+1 {
            let fx = CGFloat(x) * self.bounds.height / CGFloat(temp)
            let fx2 = CGFloat(x) * jhDraw.ARQ / CGFloat(temp) //TODO: This should be merged to fx
            let label : String = (x * intervalVaxis).description
            
            if x != temp {
                //TODO: This should be modified & deleted later on.
            drawHAxis(getX(mMargin)!, fx, self.bounds.width - getX(mMargin)!, fx)
            self.addSublayer(drawText(str: label, x: 100, y: fx2 - 600, width: jhGS.s.jhATextPanelSize, height: jhGS.s.jhATextPanelSize))
            } else { //TODO: will be chagned to the smart way
                self.addSublayer(drawText(str: label, x: 100, y: fx2 - 600, width: jhGS.s.jhATextPanelSize, height: jhGS.s.jhATextPanelSize))
            }
        }
        
        //TODO: warning guide line. There's a bug.
        if horizontalGuide {
            drawLineWithColor(mMargin, 20*mUnitOfHorizontalAxes, jhDraw.ARQ-mMargin, 20*mUnitOfHorizontalAxes, lineWidth: 2, color: UIColor(red: 254, green: 191, blue: 4).cgColor)
            drawLineWithColor(mMargin, 60*mUnitOfHorizontalAxes, jhDraw.ARQ-mMargin, 60*mUnitOfHorizontalAxes, lineWidth: 2, color: UIColor(red: 251, green: 83, blue: 96).cgColor)
        }
        
        //draw backboard
        mColor = UIColor(red: 229, green: 229, blue: 229).cgColor
        drawRect(margin: mMargin)
    }
    
    private func drawLine(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat) {
        jhDraw.worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, mLineWidth, mColor)
    }
    
    private func drawHAxis(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat) {
        jhDraw.worldLine(context: mContext, x1, y1, x2, y2, mLineWidth, mColor)
    }
    
    private func drawLineWithColor(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat, lineWidth : CGFloat, color : CGColor) {
        jhDraw.worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, lineWidth, color)
    }
    
    
    private func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * self.bounds.width / jhDraw.ARQ
        return retX
    }
    
    private func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * self.bounds.height / jhDraw.ARQ
        return retY
    }
    
    
    private func drawText(str : String, x : CGFloat, y : CGFloat, width : CGFloat, height : CGFloat) -> CALayer {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let img = renderer.image { ctx in
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .center
            let attrs = [NSAttributedStringKey.font: UIFont(name: "".font1(), size: width/2)!, NSAttributedStringKey.paragraphStyle: paragraphStyle]
            let string = str
            string.draw(with: CGRect(x: 0, y: 0, width: width, height: 10), options: .usesLineFragmentOrigin, attributes: attrs, context: nil)
        }
        //        let imageView : UIImageView = UIImageView(frame: CGRect(x: getX(x)!, y: getY(y)!, width: width, height: height))
        //
        //        imageView.image = img
        
        let tLayer = CALayer()
        let tImg = img.cgImage
        tLayer.frame = CGRect(x: getX(x)!, y: getY(y)!, width: width, height: height)
//        tLayer.frame = CGRect(x: getX(x)!, y: y - (height / 2), width: width, height: height) //TODO:
        tLayer.contents = tImg
        
        return tLayer
    }
    
    func drawRect(margin : CGFloat) {
        drawLine(margin, margin, jhDraw.ARQ-margin, margin)
        drawLine(jhDraw.ARQ-margin, margin, jhDraw.ARQ-margin, jhDraw.ARQ-margin)
        drawLine(jhDraw.ARQ-margin, jhDraw.ARQ-margin, margin, jhDraw.ARQ-margin)
        ////For DEBUG
        //        drawLine(0, 0, jhDraw.maxR, jhDraw.maxR)
        //        drawLine(0, jhDraw.maxR, jhDraw.maxR, 0)
        drawLine(margin, jhDraw.ARQ-margin, margin, margin)
    }
    
}
